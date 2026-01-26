#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace std;

const int inf = (int)2 * 1e9;
const int MOD = (int)1e9 + 7;
const int P = 239;
const int MAX_N = 200000;
const long double pi = 3.1415926;
const long double eps = 1e-6;

int main() {
  // ifstream ifs("input.in");
  // ofstream ofs("output.out");
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  string s;
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] += ('a' - 'A');
    if (s[i] == 'a' || s[i] == 'o' || s[i] == 'e' || s[i] == 'y' ||
        s[i] == 'i' || s[i] == 'u')
      continue;
    else
      cout << "." << s[i];
  }
  return 0;
}