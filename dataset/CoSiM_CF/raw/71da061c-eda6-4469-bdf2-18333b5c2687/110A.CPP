#include <bits/stdc++.h>
using namespace std;

int main() {
  int x = 0;
  string n;
  cin >> n;
  for (int i = 0; i < n.length(); i++) {
    if (n[i] == '7' || n[i] == '4')
      x++;
  }
  if (x == 7 || x == 4)
    cout << "YES";
  else
    cout << "NO";
}
