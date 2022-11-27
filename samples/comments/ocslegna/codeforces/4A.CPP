#include <bits/stdc++.h>

using namespace std;

int main() {
  int kg;
  cin >> kg;
  cout << (((kg % 2 == 0) && !(kg == 2)) ? "YES" : "NO") << endl;

  return 0;
}