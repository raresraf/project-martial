#include <bits/stdc++.h>
using namespace std;

int main() {
  string f;
  cin >> f;
  char curr;
  int maxNum;
  int currNum;
  for (int i = 0; i < f.size(); i++) {
    if (f[i] != curr) {
      maxNum = max(currNum, maxNum);
      currNum = 0;
      curr = f[i];
    } else {
      currNum++;
      maxNum = max(currNum, maxNum);
    }
  }
  if (maxNum >= 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
