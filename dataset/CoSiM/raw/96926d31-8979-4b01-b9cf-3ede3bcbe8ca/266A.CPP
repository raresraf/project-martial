#include <bits/stdc++.h>

using namespace std;

int main() {
  int a;
  cin >> a;
  char n[a];
  for (int i = 0; i < a; i++) {
    cin >> n[i];
  }
  int k = 0;
  for (int i = 0; i < a - 1; i++) {
    if (n[i] == n[i + 1])
      k++;
  }
  cout << k;
}
