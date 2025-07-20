#include <algorithm>
#include <bitset>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <istream>
#include <iterator>
#include <limits>
#include <list>
#include <locale>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ostream>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <typeinfo>
#include <utility>
#include <valarray>
#include <vector>

using namespace std;
typedef vector<int> vi;
typedef vector<string> vs;
typedef pair<int, int> pi;
typedef long long ll;
typedef unsigned long long ull;
typedef unsigned int uint;

#define sz(v) ((int)((v).size()))
#define all(v) ((v).begin(), (v).end())
#define F first
#define S second
#define PB push_back
#define MP make_pair
#define REP(i, a, b) for (int i = a; i < b; i++)
#define nl << "\n"

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  string s, v = "AOYEUIaoyeui";
  cin >> s;
  for (size_t i = 0; i < sz(s); i++) {
    if (v.find(s[i]) == string::npos)
      cout << "." << char(tolower(s[i]));
  }
  cout nl;
  return 0;
}
