// 58A - Chat room

#include <iostream>

using namespace std;

string h = "hello";
int k = 0;

void check(string s, int i) {
  int m, c;

  m = s.find(h[k], i);
  c = h.size();

  if (m >= 0) {
    k++;

    if (k < c)
      check(s, m + 1);
    if (k == c) {
      k = 0;
      cout << "YES";
    }
  } else
    cout << "NO";
}

int main() {
  string s;

  cin >> s;
  check(s, 0);

  return 0;
}
