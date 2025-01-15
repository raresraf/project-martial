#include <bits/stdc++.h>

using namespace std;

int main() {
  string inp;
  cin >> inp;
  if (inp[0] >= 'a' && inp[0] <= 'z')
    inp[0] -= 32;
  cout << inp;
}