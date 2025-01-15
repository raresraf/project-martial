#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>
using namespace std;

int main() {
  string s;
  cin >> s;
  int l = s.length();
  for (int i = 0; i < l; i++) {
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] -= 'A' - 'a';
    if (s[i] == 'a' || s[i] == 'o' || s[i] == 'y' || s[i] == 'e' ||
        s[i] == 'u' || s[i] == 'i') {
    } else {
      printf(".%c", s[i]);
    }
  }
  printf("\n");
  return 0;
}
