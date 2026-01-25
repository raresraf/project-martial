#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  int ans = 0;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    int petya, vasya, tonya;
    cin >> petya >> vasya >> tonya;
    int confidence = petya + vasya + tonya;
    (confidence >= 2) ? ++ans : ans = ans;
  }
  cout << ans;
}
