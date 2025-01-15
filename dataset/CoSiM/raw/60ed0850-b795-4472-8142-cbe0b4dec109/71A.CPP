#include <iostream>
#include <string>
using namespace std;

int main() {
  int n;
  cin >> n;
  string store[n];
  for (int i = 0; i < n; i++)
    cin >> store[i];

  for (int i = 0; i < n; i++) {
    if (store[i].length() <= 10)
      cout << store[i] << endl;
    else {
      cout << store[i].at(0) << store[i].length() - 2
           << store[i].at(store[i].length() - 1) << endl;
    }
  }
  return 0;
}