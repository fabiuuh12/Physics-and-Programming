#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace alice {

struct MemoryItem {
    int id = 0;
    std::string content;
    std::string category;
    std::string created_at;
    int use_count = 0;
};

class MemoryStore {
public:
    explicit MemoryStore(std::filesystem::path db_path);

    [[nodiscard]] const std::filesystem::path& db_path() const;
    [[nodiscard]] int count() const;
    bool add(const std::string& content, const std::string& category = "general");
    std::vector<MemoryItem> recent(int limit = 5);
    std::vector<MemoryItem> search(const std::string& query, int limit = 5);

private:
    std::filesystem::path db_path_;
    std::vector<MemoryItem> items_;
    int next_id_ = 1;

    void load();
    bool save() const;
    static std::string escape(const std::string& value);
    static std::string unescape(const std::string& value);
};

}  // namespace alice
