#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <limits>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstring>
#include <immintrin.h>
#include "MySolution.h" 

using namespace std;

static const int LOCK_MASK = 85537; 
static auto g_locks = std::make_unique<std::mutex[]>(LOCK_MASK + 1);
static std::mutex g_global_lock; 
using LockGuard = std::lock_guard<std::mutex>;

typedef float (*DistFunc)(const float*, const float*, int);
float dist_l2_scalar_safe(const float* vec1, const float* vec2, int d) {
    float res = 0;
    for (int i = 0; i < d; i++) {
        float diff = vec1[i] - vec2[i];
        res += diff * diff;
    }
    return res;
}
DistFunc g_dist_func = dist_l2_scalar_safe;

void init_SIMD() {
    g_dist_func = dist_l2_scalar_safe; 
}

inline float dist_l2_fast(const float* vec1, const float* vec2, int d) {
    return g_dist_func(vec1, vec2, d);
}

void Solution::build(int d, const vector<float>& base)
{
    static std::once_flag flag;
    std::call_once(flag, [](){ init_SIMD(); });

    finished = false; 
    
    if (M <= 0) M = 16;
    if (Mmax <= 0) Mmax = 32;
    if (Mmax0 <= 0 || Mmax0 > 100) Mmax0 = 64; 
    
    if (L <= 0) L = 16;
    if (efConstruct <= 0) efConstruct = 100;
    m_l = 1.0 / log(M); 

    dim = d;
    n_base = base.size() / dim;
    
    _base.assign(n_base, vector<float>(dim));
    for(int i = 0; i < n_base; i++) {
        std::copy(base.begin() + i * dim, base.begin() + (i + 1) * dim, _base[i].begin());
    }

    g.assign(L, vector<vector<int>>(n_base));
    layer.assign(L, vector<int>());
    
    for (int l = 0; l < L; l++) {
        int capacity = (l == 0) ? Mmax0 : Mmax; 
        for(int i = 0; i < n_base; i++) {
            g[l][i].reserve(capacity * 1.5); 
        }
    }

    int max_level_0 = (int)(-log(0.5) * m_l);
    if (max_level_0 >= L) max_level_0 = L - 1;
    for(int lc = max_level_0; lc >= 0; lc--) {
        layer[lc].push_back(0);
    }
    enterlayer = max_level_0;
    enterpoint = 0;

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::atomic<int> current_idx(1); 
    std::vector<std::thread> threads;

    auto worker = [&](int thread_id) {
        std::default_random_engine generator(std::random_device{}() + thread_id);
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        
        const int CHUNK_SIZE = 64; 
        while (true) {
            int start_idx = current_idx.fetch_add(CHUNK_SIZE);
            if (start_idx >= n_base) break;
            int end_idx = std::min(start_idx + CHUNK_SIZE, n_base);

            for (int i = start_idx; i < end_idx; i++) {
                double u = distribution(generator);
                int level = (int)(-log(u) * m_l);
                if (level >= L) level = L - 1;

                int currObj = enterpoint;
                int max_level = enterlayer; 
                
                for(int lc = max_level; lc > level; lc--) {
                    auto w = search_layer1(_base[i], currObj, 1, lc);
                    if(!w.empty()) currObj = w[0].second;
                }

                for(int lc = level; lc >= 0; lc--) {
                    Insert(currObj, i, M, Mmax, efConstruct, lc);
                }

                if (level > enterlayer) {
                    LockGuard lock(g_global_lock);
                    if (level > enterlayer) {
                        enterlayer = level;
                        enterpoint = i;
                    }
                }
            }
        }
    };
    cout<<"Finished building"<<endl;
    for (unsigned int i = 0; i < num_threads; ++i) threads.emplace_back(worker, i);
    for (auto& t : threads) if (t.joinable()) t.join();
    cout<<"exiting build"<<endl;
    finished = true;
}

float Solution::distance_square(const vector<float> &x, const vector<float> &y) {
    return dist_l2_fast(x.data(), y.data(), dim);
}

void Solution::Insert(int ep, int q, int M, int Mmax, int ef, int lc) 
{
    int real_mmax = (lc == 0) ? Mmax0 : Mmax; 
    auto W = search_layer1(_base[q], ep, ef, lc);
    auto neighbours = select_neighbour(q, W, M, lc);
    
    {
        LockGuard lock(g_locks[q & LOCK_MASK]);
        if (g[lc][q].capacity() < neighbours.size()) g[lc][q].reserve(neighbours.size() + 16);
        g[lc][q] = neighbours; 
    }
    
    static thread_local vector<pair<float, int>> candidates_buffer;

    for(auto n_id : neighbours)
    {
        LockGuard lock(g_locks[n_id & LOCK_MASK]);
        g[lc][n_id].push_back(q);
        
        if(g[lc][n_id].size() > real_mmax)
        {
            candidates_buffer.clear();
            for(int existing_n : g[lc][n_id]) {
                if (existing_n >= 0 && existing_n < n_base) {
                    float d = dist_l2_fast(_base[n_id].data(), _base[existing_n].data(), dim);
                    candidates_buffer.push_back({d, existing_n});
                }
            }
            sort(candidates_buffer.begin(), candidates_buffer.end());
            auto new_neighbors = select_neighbour(n_id, candidates_buffer, real_mmax, lc);
            g[lc][n_id] = new_neighbors;
        }
    }
}

vector<pair<float,int>> Solution::search_layer1(const vector<float> &q, int ep, int ef, int lc)
{
    static thread_local std::vector<int> tl_visited;
    static thread_local int tl_tag = 0;
    static thread_local vector<int> neighbor_buffer; 

    if (tl_visited.size() <= n_base) { 
        tl_visited.resize(n_base + 1024, 0); 
    }
    
    tl_tag++;
    if (tl_tag == 0 || tl_tag > 1000000000) { 
        std::fill(tl_visited.begin(), tl_visited.end(), 0);
        tl_tag = 1;
    }

    if(ep < 0 || ep >= n_base) return {};

    priority_queue<pair<float, int>> top_candidates; 
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> candidate_set; 

    float dist = dist_l2_fast(q.data(), _base[ep].data(), dim);
    top_candidates.push({dist, ep});
    candidate_set.push({dist, ep});
    
    tl_visited[ep] = tl_tag;
    float lower_bound = dist;

    while (!candidate_set.empty())
    {
        pair<float, int> curr = candidate_set.top();
        candidate_set.pop();

        if (curr.first > lower_bound && top_candidates.size() >= ef) break;
        
        int c_id = curr.second;
        if (c_id < 0 || c_id >= n_base) continue;

        const vector<int>* neighbors_ptr;
        
        if (!finished) {
            LockGuard lock(g_locks[c_id & LOCK_MASK]);
            neighbor_buffer = g[lc][c_id]; 
            neighbors_ptr = &neighbor_buffer;
        } else {
            neighbors_ptr = &g[lc][c_id];
            
            for(int n_id : *neighbors_ptr) {
                if (n_id >= 0 && n_id < n_base) {
                    _mm_prefetch((const char*)_base[n_id].data(), _MM_HINT_T0);
                }
            }
        }

        const vector<int>& neighbors = *neighbors_ptr;

        for (int neighbor : neighbors)
        {
            if (neighbor < 0 || neighbor >= n_base) continue;

            if (tl_visited[neighbor] != tl_tag)
            {
                tl_visited[neighbor] = tl_tag;
                float dist_n = dist_l2_fast(q.data(), _base[neighbor].data(), dim);
                
                if (top_candidates.size() < ef || dist_n < lower_bound)
                {
                    candidate_set.push({dist_n, neighbor});
                    top_candidates.push({dist_n, neighbor});
                    
                    if (top_candidates.size() > ef) {
                        top_candidates.pop(); 
                    }
                    lower_bound = top_candidates.top().first;
                }
            }
        }
    }
    vector<pair<float,int>> res;
    res.reserve(top_candidates.size());
    while(!top_candidates.empty()) {
        res.push_back(top_candidates.top());
        top_candidates.pop();
    }
    reverse(res.begin(), res.end());
    return res;
}

vector<int> Solution::select_neighbour(int q, vector<pair<float,int>>& W, int M, int lc)
{
    if(W.empty()) return {};
    vector<int> result; 
    result.reserve(M);
    
    int start_idx = 0;
    if (W.size() > 0) {
        result.push_back(W[0].second);
        start_idx = 1;
    }

    for(size_t i = start_idx; i < W.size(); i++) {
        if(result.size() >= M) break;
        
        int c_id = W[i].second;
        if (c_id < 0 || c_id >= n_base) continue;

        float dist_q_c = W[i].first;
        bool good = true;
        for(int r_id : result) {
            float dist_c_r = dist_l2_fast(_base[c_id].data(), _base[r_id].data(), dim);
            if(dist_c_r < dist_q_c) { good = false; break; }
        }
        if(good) result.push_back(c_id);
    }
    return result;
}

void Solution::search(const vector<float>& query, int* res)
{
    if (enterlayer < 0 || enterpoint < 0 || enterpoint >= n_base) { 
        fill(res, res + 10, -1); 
        return; 
    }

    int currObj = enterpoint;
    
    for (int lc = enterlayer; lc > 0; lc--) {
        auto candidates = search_layer1(query, currObj, 5, lc); 
        if (!candidates.empty()) currObj = candidates[0].second;
    }
    
    auto result_candidates = search_layer1(query, currObj, efsearch, 0);
    
    for (int i = 0; i < 10; i++) {
        if (i < result_candidates.size()) res[i] = result_candidates[i].second;
        else res[i] = -1; 
    }
}

vector<int> Solution::search_layer(const vector<float>& q, int ep, int ef, int lc) { return {}; }
int Solution::find_nearest(const vector<float>& query, const vector<int>& w) { return -1; }


void Solution::build_bf(int d, const vector<float>& base) {
    _bf_dim = d;
    _bf_base = base; 
    _bf_num = base.size() / d;
}

void Solution::search_bf(const vector<float>& query, int* res) {
    int K = 10;
    priority_queue<pair<float, int>> pq;
    for (int i = 0; i < _bf_num; i++) {
        float dist = 0;
        const float* vec_ptr = &_bf_base[i * _bf_dim];
        for (int j = 0; j < _bf_dim; j++) {
            float diff = vec_ptr[j] - query[j];
            dist += diff * diff;
        }
        if (pq.size() < K) {
            pq.push({dist, i});
        } else {
            if (dist < pq.top().first) {
                pq.pop();
                pq.push({dist, i});
            }
        }
    }
    for (int i = K - 1; i >= 0; i--) {
        if (!pq.empty()) {
            res[i] = pq.top().second;
            pq.pop();
        } else {
            res[i] = -1; 
        }
    }
}