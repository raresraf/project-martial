#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#define whatIs(x) std::cerr << #x << ": " << x << std::endl;
using namespace std;

int main() {
  long long n, k;
  cin >> n >> k;
  long long a[n + 1];
  for (auto i = 1; i <= n; ++i)
    cin >> a[i];

  long long i = 1;
  long long j = (i * (i + 1)) / 2;

  while (j < k) {
    i++;
    j = (i * (i + 1)) / 2;
  }

  i--;
  j = ((i * (i + 1)) / 2) + 1; // begin

  long long index = k - j + 1;

  cout << a[index];
}
