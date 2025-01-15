#include <iostream>
#include <queue>
#include <stack>
#include <string>
#include <vector>
using namespace std;
int main() {
  long long k, n, w, sum = 0;
  cin >> k >> n >> w;
  for (int i = 1; i <= w; i++) {
    sum = sum + i * k;
  }
  sum = sum - n;
  if (sum <= 0) {
    cout << 0 << endl;
  } else {
    cout << sum << endl;
  }
}
