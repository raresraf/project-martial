#include <iostream>

using namespace std;

int main() {
  int n;
  int a, b;
  int max = 0;
  int capacity = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    capacity += b - a;
    if (capacity > max)
      max = capacity;
  }
  cout << max << endl;
  return 0;
}
