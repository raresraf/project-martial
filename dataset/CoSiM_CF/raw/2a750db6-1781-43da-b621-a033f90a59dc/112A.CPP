// https://codeforces.com/problemset/problem/112/A

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctype.h>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

using namespace std;

int main() {

  string a, b;
  cin >> a >> b;
  int n = a.length();
  vector<int> m, k;
  int s1 = 0, s2 = 0;
  transform(a.begin(), a.end(), a.begin(), ::toupper);
  transform(b.begin(), b.end(), b.begin(), ::toupper);
  copy(a.begin(), a.end(), back_inserter(m));
  copy(b.begin(), b.end(), back_inserter(k));

  for (int i = 0; i < n; i++) {
    s1 += m[i];
    s2 += k[i];
    if (s1 != s2)
      break;
  }

  if (s1 > s2)
    cout << "1";

  else if (s2 > s1)
    cout << "-1";

  else
    cout << "0";

  return 0;
}