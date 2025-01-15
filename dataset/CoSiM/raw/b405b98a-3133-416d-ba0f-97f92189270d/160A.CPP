#include <algorithm>
#include <iostream>
#include <vector>
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);
using namespace std;
int main() {
  optimizar_io long long a, sum = 0, sum1 = 0, cont = 0;
  cin >> a;
  vector<long long> v(a);
  for (int i = 0; i < a; ++i) {
    cin >> v[i];
    sum += v[i];
  }
  sum /= 2;
  sort(v.begin(), v.end());
  for (int i = a - 1; i >= 0; --i) {
    sum1 += v[i];
    cont++;
    if (sum1 > sum)
      break;
  }
  cout << cont << "\n";
  return 0;
}
