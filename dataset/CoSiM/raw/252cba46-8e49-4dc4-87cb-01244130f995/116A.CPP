#include <iostream>
#include <string>

using namespace std;

int main() {

  int n;
  cin >> n;
  int capacity = 0;
  int max = 0;
  int exit, enter;

  for (int i = 0; i < n; i++) {
    cin >> exit >> enter;
    capacity = capacity - exit + enter;
    if (capacity > max)
      max = capacity;
  }
  cout << max << endl;
  return 0;
}
