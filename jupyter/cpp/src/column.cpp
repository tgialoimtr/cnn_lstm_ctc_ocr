#include "column.h"
#include <algorithm>

int add(int i, int j)
{
    return i + j + 1;
}

int subtract(int i, int j)
{
    return i - j;
}

int min(int a, int b, int c) {
    int result = a;
    if (result > b) result = b;
    if (result > c) result = c;
    return result;
}


TemplateFilterer::TemplateFilterer()
{
}

void TemplateFilterer::add(std::string code, const std::vector<std::string>& mallkws, const std::vector<std::string>& storekws)
{
    for (const auto& kw : mallkws)
    {
        auto iter = m_mallkws.find(kw);
        if (iter == m_mallkws.end())
        {
            m_mallkws[kw] = std::set<std::string>({code});
        }
        else
        {
            iter->second.insert(code);
        }
    }
    for (const auto& kw : storekws)
    {
        auto iter = m_storekws.find(kw);
        if (iter == m_storekws.end())
        {
            m_storekws[kw] = std::set<std::string>({code});
        }
        else
        {
            iter->second.insert(code);
        }
    }

}

TemplateFilterer::Result TemplateFilterer::match(const std::string& text, const std::string& pattern)
{
    const auto& t = text.data();
    const auto& p = pattern.data();
    int n = text.length();
    int m = pattern.length();

    TemplateFilterer::Result result;

    int **g = new int*[m+1];
    for(int i = 0; i < m+1; ++i) {
        g[i] = new int[n+1];
    }
               
               
    int distance = 999;
    int pos = 0;
    for (int i = 0; i < m + 1; ++i) {
        g[i][0] = i;
        for (int j = 1; j < n + 1; ++j) {
            g[i][j] = 0;
        }
    }   
    for (int j = 1; j < n + 1; ++j) {
        for (int i = 1; i < m + 1; ++i) {
            int delta = 1;
            if (t[j - 1] == p[i - 1]) {
                delta = 0;
            }
            g[i][j] = min(g[i - 1][j - 1] + delta,
                            g[i - 1][j] + 1,
                            g[i][j - 1] + 1);
        }
        if (g[m][j] <= distance) {
            distance = g[m][j];
            pos = j;
        }
    }
 
    for(int i = 0; i < m+1; ++i) {
        delete [] g[i];
    }
    delete [] g;

    result.len = m;
    result.distance = distance;
    result.match_ratio = 1.0*(m - distance)/m;
    result.match = (result.match_ratio > 0.8f);
    result.start_pos = pos - m;

    return result;
}


std::vector<std::string> TemplateFilterer::search(const std::string& alltexts)
{
    
    std::set<std::string> matched_mall;
    std::set<std::string> matched_store;
    for (const auto& item: m_mallkws)
    {
        Result rs = match(alltexts, item.first);
        if (rs.match)
        {
            // std::cout<<item.first<<" from " << alltexts.substr(rs.start_pos, rs.len) << " with distance " << rs.distance <<std::endl;
            matched_mall.insert(item.second.begin(), item.second.end());
        }
    }
    for (const auto& item: m_storekws)
    {
        Result rs = match(alltexts, item.first);
        if (rs.match)
        {
            // std::cout<<item.first<<" from " << alltexts.substr(rs.start_pos, rs.len) << " with distance " << rs.distance <<std::endl;
            matched_store.insert(item.second.begin(), item.second.end());
        }
    }
    // std::cout<<std::endl;
    // std::cout<<alltexts;
    // std::cout<<std::endl;

    // for (const auto& code: matched_mall)
    // {
    //     std::cout<<"mall "<<code;
    // }

    // for (const auto& code: matched_store)
    // {
    //     std::cout<<"store "<<code;
    // }

    // intersect
    std::vector<std::string> ret;
    std::set_intersection(matched_mall.begin(),matched_mall.end(),matched_store.begin(),matched_store.end(), std::back_inserter(ret));

    // for (const auto& code: ret)
    // {
    //     std::cout<<"ret "<<code;
    // }

    return ret;
}


int main () {
    TemplateFilterer tf = TemplateFilterer();
    
    tf.add("L1", std::vector<std::string>({ "JUNTION8" , "9 Bishan Place"})  ,std::vector<std::string>({  "FOUR LEAVES","M2-12345-678" }));
    tf.add("L2", std::vector<std::string>({ "BUSGIS JUNCTION" , "200 Victoria"})  ,std::vector<std::string>({  "hello", "wassup" }));
    tf.add("L3", std::vector<std::string>({})  ,std::vector<std::string>({  "FOUR LEAVES","M2-12345-678"}));

    for (const auto& item: tf.m_storekws)
    {
        std::cout<<std::endl;
        std::cout<<item.first<<": ";
        for (const auto& code: item.second)
        {
            std::cout<<code;
        }
        std::cout<<std::endl;
    }

    const char* alltexts = "Hi BUSGIS JUNCTION, #K11-12\n"
                            "FOUR LEAVES FOUR LEAVES\n"
                            "wassup chau len ba\n"
                            "FOUR LEAFES";
    std::vector<std::string> ret = tf.search(std::string(alltexts));
    return 0;
}


