#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, k, sum = 0;
  cin >> n >> k;
  vector<int> array1(n, 0);

  for (int i = 0; i < n; i++) {
    cin >> array1[i];
  }

  for (int i = 0; i < n; i++) {
    if (array1[i] >= array1[k - 1] && array1[i] > 0) {
      sum += 1;
    }
  }
  cout << sum << endl;
  return 0;
}