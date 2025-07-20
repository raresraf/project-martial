#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, k, count = 0;
  int x[100], target;
  cin >> n >> k;
  k--;
  for (int i = 0; i < n; i++) {
    int a;
    cin >> a;
    x[i] = a;
  }
  for (int i = 0; i < n; ++i) {
    if (x[i] >= x[k] && x[i] > 0) {
      count++;
    }
  }

  std::cout << count << endl;
}