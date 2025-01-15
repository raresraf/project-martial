#include <iostream>
using namespace std;

int main() {

  int n, count = 0;
  cin >> n;
  char cl[n];

  for (int i = 0; i < n; i++) {
    cin >> cl[i];
  }
  for (int j = 0; j < n; j++) {
    if (cl[j] != cl[j + 1]) {
      count++;
    }
  }
  cout << n - count << endl;

  return 0;
}
