#include <iostream>

using namespace std;
int main() {
  int n, numberDiff;
  cin >> n;
  numberDiff = 0;

  char stones[55];
  for (int i = 0; i < n; i++) {
    cin >> stones[i];
  }

  for (int i = 0; i < n; i++) {
    if (stones[i] == stones[i + 1]) {
      numberDiff++;
    } else {
    }
  }
  cout << numberDiff;
}
