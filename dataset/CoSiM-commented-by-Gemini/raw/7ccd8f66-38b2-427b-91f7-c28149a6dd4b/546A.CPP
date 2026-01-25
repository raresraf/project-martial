#include <iostream>
using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;

  int total = 0;
  for (int i = 1; i <= w; i++)
    total += i * k;

  int result = total - n;
  if (result < 0)
    result = 0;
  cout << result << endl;

  return 0;
}