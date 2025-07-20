#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <utility>
#include <vector>

#define clr(name, val) memset(name, (val), sizeof(name));
#define EPS .000000001
#define ll long long
#define psb(b) push_back((b))
#define ppb() pop_back()
#define oo 10000000
#define swap(x, y, t)                                                          \
  {                                                                            \
    (t) = (x);                                                                 \
    (x) = (y);                                                                 \
    (y) = (t);                                                                 \
  }
#define for_i(s, n) for (int i = (s); i < (n); (i)++)
#define for_j(s, n) for (int(j) = (s); (j) < (n); (j)++)
#define for_k(s, n) for (int(k) = (s); (k) < (n); (k)++)
#define MAX

/// next_permutation  next_permutation (s.begin(),s.end())
/// reverse(a,a+n);
/// binary_search(first,last);
/// vector erase v.erase(v.begin()+position);
/// map map<int , int > data;
/// map clear  data.clear();
/// map iterator>>>> map <int,vector <int> >::const_iterator it;
/// find an element in map (colour.find(nd)==colour.end());//if it return true
/// this mean the element is'nt in the map. pass a vector to a funtion: funtion
/// (vector <data type> &vector name); make_pair  point=make_pair(i,j); access
/// pair value point.first;point.second;

using namespace std;

/// int rr[]= {-1,-1,0,0,1,1};
/// int cc[]= {-1,0,-1,1,0,1};
/// int rr[]= {0,0,1,-1};/*4 side move*/
/// int cc[]= {-1,1,0,0};/*4 side move*/
/// int rr[]= {1,1,0,-1,-1,-1,0,1};/*8 side move*/
/// int cc[]= {0,1,1,1,0,-1,-1,-1};/*8 side move*/
/// int rr[]={1,1,2,2,-1,-1,-2,-2};/*night move*/
/// int cc[]={2,-2,1,-1,2,-2,1,-1};/*night move*/

void input(string &p, char ch) {
  while (scanf("%c", &ch) == 1) {
    if (ch == '\n')
      break;
    p += tolower(ch);
  }
  return;
}

int main() {
  string a, b;
  char ch;
  input(a, ch);
  input(b, ch);
  if (a > b)
    cout << 1 << endl;
  else if (a == b)
    cout << 0 << endl;
  else
    cout << -1 << endl;
  return 0;
}
