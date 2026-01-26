#include <bits/stdc++.h>
using namespace std;

int main() {
  string a;
  cin >> a;
  int b, i, d, e, c = 0;
  vector<int> p;
  for (i = 0; i < a.size(); i++) {
    b = a[i];
    p.push_back(b);
  }
  vector<int> q;
  char t;
  for (i = 0; i < a.size(); i++) {
    if (p[i] == 65 || p[i] == 69 || p[i] == 73 || p[i] == 79 || p[i] == 85 ||
        p[i] == 89 || p[i] == 97 || p[i] == 101 || p[i] == 105 || p[i] == 117 ||
        p[i] == 111 || p[i] == 121) {
      c = c + 0;
    } else if (p[i] <= 90 && p[i] >= 65) {
      p[i] = p[i] + 32;
      d = 46;
      q.push_back(d);
      q.push_back(p[i]);
      c = c + 2;
    } else {
      d = 46;
      q.push_back(d);
      q.push_back(p[i]);
      c = c + 2;
    }
  }
  for (i = 0; i < c; i++) {
    t = q[i];
    cout << t;
  }
}