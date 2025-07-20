#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n;
  cin >> n;
  int sum, count = 0;
  vector<int> array1(n, 0);
  while (n > 0) {
    sum = 0;
    for (int i = 0; i < 3; i++) {
      cin >> array1[i];
      if (array1[i] == 1)
        sum++;
    }
    if (sum >= 2)
      count++;
    n--;
  }
  cout << count << endl;
  return 0;
}