#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;

  cin >> n;

  for (int i = 1; i <= n; i++) {
    string name;
    cin >> name;
    int len = name.length();
    if (len <= 10)
      cout << name << endl;
    else {
      cout << name[0] << len - 2 << name[len - 1] << endl;
    }
  }

  return 0;
}
