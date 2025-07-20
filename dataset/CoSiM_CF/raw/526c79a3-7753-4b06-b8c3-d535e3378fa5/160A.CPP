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

int a[100005];
int main() {
  int p;
  cin >> p;
  int i;
  for (i = 0; i < p; ++i) {
    /* code */
    cin >> a[i];
  }
  sort(a, a + p);
  int ck = 0, meyer = 0, nibe = 0;
  // int i;
  for (i = p - 1; i >= 0; i--) {
    /* code */
    nibe++;
    ck += a[i];
    meyer = 0;
    for (int j = 0; j < i; j++) {
      meyer += a[j];
      // cout << ck << " " << meyer << endl;
      if (meyer > ck) {
        break;
      }
    }

    if (ck > meyer)
      break;
    // cout << ck << " " << meyer << endl;
  }
  cout << nibe << endl;
}