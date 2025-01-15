#include <iostream>

using namespace std;

int main() {
  int x, y;
  cin >> x >> y;

  int arr[50], count = 0;

  for (auto i = 0; i < x; i++) {
    cin >> arr[i];
  }

  for (auto i = 0; i < x; i++) {
    if (arr[i] == 0)
      continue;
    if (arr[i] >= arr[y - 1])
      count++;
  }

  cout << count << endl;

  return 0;
}
