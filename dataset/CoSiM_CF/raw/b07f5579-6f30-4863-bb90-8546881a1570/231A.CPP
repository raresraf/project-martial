#include <iostream>

using namespace std;

int main() {
  int n, arr[1000], count = 0, total = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    for (int i = 0; i < 3; i++) {
      cin >> arr[i];
      total += arr[i];
    }
    if (total >= 2)
      count += 1;
    total = 0;
  }

  cout << count << endl;

  return 0;
}