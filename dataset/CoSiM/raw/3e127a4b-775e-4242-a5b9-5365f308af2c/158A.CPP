#include <iostream>
using namespace std;
int main() {
  int n, k, x, count = 0;
  cin >> n >> k;
  for (int i = 0; i < n; i++) {
    int temp;
    cin >> temp;
    if (temp > 0) {
      if (i < k || temp == x)
        count++;
    }
    if (i + 1 == k) {
      x = temp;
    }
  }
  cout << count;
}