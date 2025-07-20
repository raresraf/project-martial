#include <algorithm>
#include <bits/stdc++.h>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
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
int main() {
  string s;
  cin >> s;

  for (int i = 0; i < s.size(); i++) {
    char c = towlower(s[i]);
    if (c == 'a' || c == 'o' || c == 'y' || c == 'e' || c == 'u' ||
        c == 'i') // Vowels are letters "A", "O", "Y", "E", "U", "I"
    {
      continue;
    } else {
      cout << '.' << c;
    }
  }
  return 0;
}
