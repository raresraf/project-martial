#include <iostream>

using namespace std;

int main() {
  int n, sol[3], solVal = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    int sum = 0;
    for (int j = 0; j < 3; j++) {
      cin >> sol[j];
      sum += sol[j];
      if (j == 2 && sum >= 2)
        solVal++;
    }
  }
  cout << solVal;
  return 0;
}