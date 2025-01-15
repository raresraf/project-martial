#include <iostream>

using namespace std;
int main() {
  int t, array[3], num = 0;
  cin >> t;
  for (; t > 0; t--) {
    cin >> array[0] >> array[1] >> array[2];
    if (array[0] + array[1] + array[2] >= 2) {
      num++;
    }
  }
  cout << num << endl;
  return 0;
}
