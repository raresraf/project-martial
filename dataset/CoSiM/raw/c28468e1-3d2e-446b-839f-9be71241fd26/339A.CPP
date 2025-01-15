#include <iostream>
#include <vector>
using namespace std;
int main() {

  string n;
  vector<int> arr;
  cin >> n;

  for (int i = 0; i < n.length(); i++) {
    if (isdigit(n[i])) {
      arr.push_back(n[i] - 48);
    }
  }

  for (int i = 0; i < arr.size() - 1; i++) {
    for (int j = i + 1; j < arr.size(); j++) {
      if (arr[i] > arr[j]) {
        int sv = arr[i];
        arr[i] = arr[j];
        arr[j] = sv;
      }
    }
  }

  for (int i = 0; i < arr.size(); i++) {
    cout << arr[i];
    if (i != arr.size() - 1) {
      cout << "+";
    }
  }

  return 0;
}
