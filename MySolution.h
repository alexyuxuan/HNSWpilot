#ifndef MYSOLUTION_H
#define MYSOLUTION_H
#include<bits/stdc++.h>
using namespace std;

class Solution
{
    public:
        vector<int> visited;
        int tag = 1;

        void search_batch(const vector<vector<float>>& queries, vector<vector<int>>& results);
        int dim=128,n_base,L=16,M=32,Mmax = 50, Mmax0 = 256, efConstruct = 500,enterpoint,enterlayer,
        efsearch = 200;
        bool finished = false;
        double m_l;
        
        void build(int d,const vector<float>& base);
        void search(const vector<float>& query, int*res);
        
        void Insert(int ep,int q, int M, int Mmax,int ef, int lc); //ef是每个店粗筛的个数
        vector<int> select_neighbour(int q,vector<pair<float,int>> &W, int M, int lc);
        // bool cmp(int i, int j);
        
        
        float distance_square(const vector<float>& x,const vector<float>& y);
        int find_nearest(const vector<float>& query, const vector<int>& w);
        vector<int> search_layer(const vector<float>& q, int ep, int ef, int lc);
        vector<pair<float,int>> search_layer1(const vector<float>& q, int ep, int ef, int lc);

        vector<vector<float>> _base;
        vector<vector<vector<int>>> g;
        vector<vector<int>> layer;



        void build_bf(int d, const vector<float>& base);
        void search_bf(const vector<float>& query, int* res);
    
         vector<float> _bf_base; // 存储暴力搜索的底库数据
    int _bf_dim;            // 维度
    int _bf_num;  

};
#endif