#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

using namespace std;

// #include <ext/pb_ds/assoc_container.hpp>
// #include <ext/pb_ds/tree_policy.hpp>
// using namespace __gnu_pbds;
// typedef tree<int,null_type,less<int>,rb_tree_tag,
// tree_order_statistics_node_update> ordered_set;

int main(int argc, char const *argv[]) {
  int a;
  cin >> a;
  long long ans = 0;
  for (int i = 2; i < a; ++i) {
    ans += i * (i + 1);
  }
  cout << ans << endl;
  return 0;
}