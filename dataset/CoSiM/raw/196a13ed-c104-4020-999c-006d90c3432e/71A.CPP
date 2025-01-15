#include <cmath>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string arr[100];
  int n;

  cin >> n;

  for (int i = 0; i < n; i++) {

    cin >> arr[i];
  }
  for (int i = 0; i < n; i++) {
    if (arr[i].length() <= 10) {

      cout << arr[i] << endl;
    } else if (arr[i].length() > 10) {
      cout << arr[i][0] << arr[i].length() - 2 << arr[i][arr[i].length() - 1]
           << endl;
    }
  }
}