#include <iostream>
using namespace std;
int main() {
  long long int n, w, k, m = 0;
  cin >> k;
  cin >> n;
  cin >> w;
  m = (w * (w + 1)) / 2;
  m = m * k;
  if (n <= m)
    cout << m - n;
  else
    cout << 0;
}