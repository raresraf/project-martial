// Implementation.

#include <bits/stdc++.h>
#include <cmath>

using namespace std;

int main(void) {
  int n, count = 0;
  char s, prev;
  cin >> n;
  while (n--) {
    cin >> s;
    if (s == prev) {
      count++;
    } else {
      prev = s;
    }
  }
  cout << count << endl;
  return 0;
}