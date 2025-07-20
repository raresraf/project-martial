#include <iostream>
using namespace std;
#define soads_game int main()

soads_game {
  int i, j, k;
  long long sum = 0;
  cin >> i >> j >> k;
  for (int x = 1; x <= k; x++) {
    sum += i * x;
  }
  if (sum >= j)
    cout << sum - j;
  else
    cout << "0";
  return 0;
}
