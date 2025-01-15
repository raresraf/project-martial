#include <bits/stdc++.h>
using namespace std;

int main() {
  string fs, ss; // fs=first string , ss=second string
  cin >> fs;
  cin >> ss;
  for (int i = 0; i < fs.size(); i++) {
    if (fs[i] > 64 && fs[i] < 91) // ASCII Value of A=65 and Z=90
                                  // again, ASCII Value of a=97 and z=122
    {
      fs[i] += 32;
    }
    if (ss[i] > 64 && ss[i] < 91) {
      ss[i] += 32;
    }
  }
  if (fs == ss) {
    cout << 0;
  } else if (fs > ss) {
    cout << 1;
  } else if (fs < ss) {
    cout << -1;
  }
  return 0;
}
