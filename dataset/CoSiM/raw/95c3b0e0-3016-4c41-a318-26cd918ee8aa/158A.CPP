#include <iostream>
using namespace std;
int main() {
  int n, k, data[100];
  int count = 0;
  cin >> n >> k;
  for (int i = 0; i < n; i++) {
    cin >> data[i];
  }
  for (int i = 0; i < n; i++) {
    if (data[i] > 0 && data[i] >= data[k - 1])
      count++;
  }
  cout << count;
  return 0;
}
