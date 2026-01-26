#include <iostream>
using namespace std;
int main() {
  int n;
  int arr[3];
  cin >> n;
  int count = 0;
  int total = 0;
  while (n--) {
    for (int i = 0; i < 3; i++) {
      cin >> arr[i];
      if (arr[i] == 1) {
        count++;
      }
    }
    if (count >= 2) {
      total++;
    }
    count = 0;
  }
  cout << total << endl;

  return 0;
}