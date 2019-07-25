#include <list>
#include <unordered_map>
#include <set>
#include <vector>
#include <iostream>

/*! Add two integers
    \param i an integer
    \param j another integer
*/
int add(int i, int j);
/*! Subtract one integer from another 
    \param i an integer
    \param j an integer to subtract from \p i
*/
int subtract(int i, int j);

using namespace std;

struct TemplateFilterer
{
    struct Result
    {
        bool match;
        float match_ratio;
        int len;
        int start_pos;
        int distance;
    };

    TemplateFilterer();
    Result match(const string& line, const string& pattern);

    unordered_map<string, set<string> > m_mallkws;
    unordered_map<string, set<string> > m_storekws;



    void add(string code, const vector<string>& mallkws, const vector<string>& storekws);
    vector<string> search(const string& alltexts);

};
