#include <iostream>
using namespace std;
int main() {
  int n, i;
  cin >> n;
  int a[1000], b[1000];
  for (i = 0; i < n; i++) {
    cin >> a[i] >> b[i];
  }
  int cumulative, max;
  cumulative = a[0] + b[0];
  max = cumulative;
  for (i = 1; i < n; i++) {
    cumulative = cumulative + b[i] - a[i];
    if (cumulative > max)
      max = cumulative;
  }
  cout << max << endl;
  return 0;
}
