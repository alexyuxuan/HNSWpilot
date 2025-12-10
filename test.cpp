#include "MySolution.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <random>
#include <cmath>

// 模拟项目中的read_data函数 - 修改为只读取前1000个数据
void read_data(int cas, std::vector<float> &base_, int &dimension_, int &pointEnum)
{
    std::string filePath;

    if (cas == 0)
    {
        filePath = "glove_base.txt";
        // pointEnum = 1183514; // 只测试1000个数据
        pointEnum = 100000; // 只测试1000个数据

        dimension_ = 100;
    }
    else if (cas == 1)
    {
        filePath = "sift_base.txt";
        pointEnum = 100000; // 测试的数据个数
        dimension_ = 128;
    }
    else
    {
        std::cerr << "Invalid dataset case: " << cas << std::endl;
        return;
    }

    base_.resize(dimension_ * pointEnum);
    std::ifstream fin(filePath);
    if (!fin.is_open())
    {
        std::cerr << "Error opening " << filePath << std::endl;
        return;
    }
    std::ios::sync_with_stdio(false); // 禁用同步
    fin.tie(NULL);                    // 解绑流

    for (int i = 0; i < pointEnum; ++i)
    {
        for (int j = 0; j < dimension_; ++j)
        {
            int index = i * dimension_ + j;
            if (!(fin >> base_[index]))
            {
                std::cerr << "Error reading data at point " << i << ", dimension " << j << std::endl;
                fin.close();
                return;
            }
        }
    }
    fin.close();

    std::cout << "Loaded dataset case " << cas << ": " << filePath << std::endl;
    std::cout << "Vectors: " << pointEnum << ", Dimension: " << dimension_ << std::endl;
}

// 生成随机查询向量
void generate_query_vectors(int dimension_, int query_count, std::vector<std::vector<float>> &query_vectors)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    query_vectors.resize(query_count, std::vector<float>(dimension_));
    for (int i = 0; i < query_count; i++)
    {
        for (int j = 0; j < dimension_; j++)
        {
            query_vectors[i][j] = dist(gen);
            // std::cout << query_vectors[i][j] << " ";
        }
        // std::cout << std::endl;
    }
}

// 计算Top-K准确率
float compute_recall(const std::vector<int> &bf_res, const std::vector<int> &hnsw_res, int K)
{
    int hit = 0;
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            if (bf_res[i] == hnsw_res[j])
            {
                hit++;
                break;
            }
        }
    }
    return float(hit) / K;
}

// 从数据集中随机选择查询向量（从前1000个中选）
void generate_query_from_data(const std::vector<float> &base_, int dimension_, int query_count,
                              std::vector<std::vector<float>> &query_vectors)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    int total_points = base_.size() / dimension_;
    std::uniform_int_distribution<int> dist(0, total_points - 1);

    query_vectors.resize(query_count, std::vector<float>(dimension_));
    for (int i = 0; i < query_count; i++)
    {
        int point_idx = dist(gen);
        for (int j = 0; j < dimension_; j++)
        {
            query_vectors[i][j] = base_[point_idx * dimension_ + j];
        }
    }
}

// 测试指定数据集
void test_dataset_case(int cas, const std::string &dataset_name)
{
    std::cout << "\n=== Testing Dataset: " << dataset_name << " (Case " << cas << ") ===" << std::endl;
    // std::cout << "Note: Using only first 1000 data points for testing" << std::endl;

    // 读取数据
    std::vector<float> base_;
    int dimension_, pointEnum;
    read_data(cas, base_, dimension_, pointEnum);

    if (base_.empty())
    {
        std::cout << "Data loading failed, skipping test" << std::endl;
        return;
    }

    // 创建查询向量 - 从实际数据f中随机选择
    const int QUERY_COUNT = 1000; // 减少查询数量以加快测试
    const int K = 10;
    std::vector<std::vector<float>> query_vectors;
    generate_query_from_data(base_, dimension_, QUERY_COUNT, query_vectors);
    // generate_query_vectors(dimension_, QUERY_COUNT, query_vectors);

    // ========== Brute Force Search ==========
    std::cout << "\n--- Brute Force Search Test ---" << std::endl;
    Solution bf_solution;
    auto bf_build_start = std::chrono::high_resolution_clock::now();
    bf_solution.build_bf(dimension_, base_);
    auto bf_build_end = std::chrono::high_resolution_clock::now();
    double bf_build_time = std::chrono::duration<double>(bf_build_end - bf_build_start).count();
    std::cout << "[BF] Index build time: " << bf_build_time << " seconds" << std::endl;

    double bf_total_time = 0.0;
    uint64_t bf_total_dist_count = 0;
    std::vector<std::vector<int>> bf_results(QUERY_COUNT, std::vector<int>(K));

    for (int i = 0; i < QUERY_COUNT; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        bf_solution.search_bf(query_vectors[i], bf_results[i].data());
        auto end = std::chrono::high_resolution_clock::now();
        bf_total_time += std::chrono::duration<double>(end - start).count();

        if (i % 100 == 0)
        {
            std::cout << "BF query " << i << " completed" << std::endl;
        }
    }

    double bf_avg_time = bf_total_time / QUERY_COUNT;
    std::cout << "[BF] Average search time: " << bf_avg_time << " seconds, QPS: " << 1.0 / bf_avg_time << std::endl;
    std::cout << "[BF] Average distance computations per query: " << (double)bf_total_dist_count / QUERY_COUNT << std::endl;

    // ========== HNSW Search ==========
    std::cout << "\n--- HNSW Search Test ---" << std::endl;
    Solution hnsw_solution;
    auto hnsw_build_start = std::chrono::high_resolution_clock::now();
    hnsw_solution.build(dimension_, base_);
    auto hnsw_build_end = std::chrono::high_resolution_clock::now();
    double hnsw_build_time = std::chrono::duration<double>(hnsw_build_end - hnsw_build_start).count();
    std::cout << "[HNSW] Index build time: " << hnsw_build_time << " seconds" << std::endl;

    double hnsw_total_time = 0.0;
    uint64_t hnsw_total_dist_count = 0;
    std::vector<std::vector<int>> hnsw_results(QUERY_COUNT, std::vector<int>(K));

    for (int i = 0; i < QUERY_COUNT; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        hnsw_solution.search(query_vectors[i], hnsw_results[i].data());
        auto end = std::chrono::high_resolution_clock::now();
        hnsw_total_time += std::chrono::duration<double>(end - start).count();

        if (i % 100 == 0)
        {
            std::cout << "HNSW query " << i << " completed" << std::endl;
        }
    }

    double hnsw_avg_time = hnsw_total_time / QUERY_COUNT;
    std::cout << "[HNSW] Average search time: " << hnsw_avg_time << " seconds, QPS: " << 1.0 / hnsw_avg_time << std::endl;
    std::cout << "[HNSW] Average distance computations per query: " << (double)hnsw_total_dist_count / QUERY_COUNT << std::endl;

    // ========== Calculate Top-K Accuracy ==========
    std::cout << "\n--- Accuracy Evaluation ---" << std::endl;
    double recall_sum = 0.0;
    for (int i = 0; i < QUERY_COUNT; ++i)
    {
        float recall = compute_recall(bf_results[i], hnsw_results[i], K);
        recall_sum += recall;
        if (i % (QUERY_COUNT / 10) == 0)
        {
            std::cout << "Query " << i << " recall: " << recall << std::endl;
        }
    }
    double avg_recall = recall_sum / QUERY_COUNT;
    std::cout << "Top-" << K << " Average recall (HNSW vs BF): " << avg_recall << std::endl;

    // 显示示例查询结果
    std::cout << "\n--- Example Query Results Comparison ---" << std::endl;
    for (int example = 0; example < std::min(5, QUERY_COUNT); example++)
    {
        std::cout << "Example query " << example << ":" << std::endl;
        std::cout << "HNSW results: ";
        for (int i = 0; i < K; i++)
            std::cout << hnsw_results[example][i] << " ";
        std::cout << "\nBF results:   ";
        for (int i = 0; i < K; i++)
            std::cout << bf_results[example][i] << " ";
        std::cout << std::endl;

        // 计算该查询的准确率
        float example_recall = compute_recall(bf_results[example], hnsw_results[example], K);
        std::cout << "Query recall: " << example_recall << std::endl
                  << std::endl;
    }

    std::string filePath1 = "query.txt";
    std::ifstream fin1(filePath1);
    if (!fin1.is_open())
    {
        std::cerr << "Error opening " << filePath1 << std::endl;
        return;
    }
    std::ios::sync_with_stdio(false);
    fin1.tie(nullptr);

    std::string filePath2 = "truth.txt";
    std::ifstream fin2(filePath2);
    if (!fin2.is_open())
    {
        std::cerr << "Error opening " << filePath2 << std::endl;
        return;
    }
    std::ios::sync_with_stdio(false);
    fin2.tie(nullptr);

    std::cout << "\n--- HNSW Search Test on sift_query.txt ---" << std::endl;
    std::vector<std::vector<float>> queries;
    const int dimension = 100;
    const int query_count = 10000;
    queries.resize(query_count, std::vector<float>(dimension));
    for (int i = 0; i < query_count; ++i)
    {
        for (int j = 0; j < dimension; ++j)
        {
            fin1 >> queries[i][j];
        }
    }

    std::vector<std::vector<int>> groundtruth;
    // calculate top 10 recall
    // calculate search time
    int groundtruth_size = 100;
    groundtruth.resize(query_count, std::vector<int>(K));
    for (int i = 0; i < query_count; ++i)
    {
        for (int j = 0; j < groundtruth_size; ++j)
        {
            int val;
            fin2 >> val;
            if (j < K)
            {
                groundtruth[i][j] = val;
            }
        }
    }
    fin1.close();
    fin2.close();

    std::vector<std::vector<int>> hnsw_res(query_count, std::vector<int>(K));
    // Reset timing for the new sift_query.txt evaluation
    double hnsw_total_time_sift = 0.0;
    uint64_t hnsw_total_dist_count_sift = 0;
    for (int i = 0; i < query_count; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        hnsw_solution.search(queries[i], hnsw_res[i].data());
        auto end = std::chrono::high_resolution_clock::now();
        hnsw_total_time_sift += std::chrono::duration<double>(end - start).count();

        if (i % 1000 == 0)
        {
            std::cout << "HNSW query " << i << " completed" << std::endl;
        }
    }
    double hnsw_avg_time2 = hnsw_total_time_sift / query_count;
    std::cout << "[HNSW] Average search time on sift_query.txt: " << hnsw_avg_time2 << " seconds, QPS: " << 1.0 / hnsw_avg_time2 << std::endl;
    std::cout << "[HNSW] Average distance computations per query on sift: " << (double)hnsw_total_dist_count_sift / query_count << std::endl;

    double recall_sum2 = 0.0;
    for (int i = 0; i < query_count; ++i)
    {
        float recall = compute_recall(groundtruth[i], hnsw_res[i], K);
        recall_sum2 += recall;
    }
    double avg_recall2 = recall_sum2 / query_count;
    std::cout << "Top-" << K << " Average recall (HNSW vs Ground Truth): " << avg_recall2 << std::endl;
}

// 检查文件是否存在
bool check_file_exists(const std::string &filename)
{
    std::ifstream file(filename);
    return file.good();
}

int main()
{
    std::cout << "Vector Search Algorithm Performance Comparison Test" << std::endl;
    std::cout << "===================================================" << std::endl;
    // std::cout << "Note: Using only first 1000 data points for testing" << std::endl;

    bool has_real_data = false;

    // Test GLOVE dataset
    if (check_file_exists("glove_base.txt"))
    {
        test_dataset_case(0, "GLOVE");
        has_real_data = true;
    }
    else
    {
        std::cout << "\nGLOVE dataset file not found: glove_base.txt" << std::endl;
    }

    // Test SIFT dataset
    // if (check_file_exists("sift_base.txt"))
    // {
    //     test_dataset_case(0, "SIFT");
    //     has_real_data = true;
    // }
    // else
    // {
    //     std::cout << "\nSIFT dataset file not found: sift_base.txt" << std::endl;
    // }

    if (!has_real_data)
    {
        std::cout << "\nNo valid dataset available for testing!" << std::endl;
        std::cout << "Please ensure sift_base.txt file exists in current directory" << std::endl;
    }

    std::cout << "\nTest completed!" << std::endl;
    return 0;
}