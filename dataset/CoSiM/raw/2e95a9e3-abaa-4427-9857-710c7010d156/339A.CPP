#include <algorithm>
#include <iostream>
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);
using namespace std;
int main() {
  optimizar_io string a, b = "";
  cin >> a;
  for (int i = 0; i < (int)a.size(); ++i)
    if (a[i] != '+')
      b += a[i];
  sort(b.begin(), b.end());
  for (int i = 0; i < (int)b.size() - 1; ++i)
    cout << b[i] << "+";
  cout << b[b.size() - 1] << "\n";
  return 0;
}
