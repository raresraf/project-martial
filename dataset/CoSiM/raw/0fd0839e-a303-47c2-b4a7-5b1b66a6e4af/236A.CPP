#include <iostream>
#include <vector>
using namespace std;
int main() {

  vector<int> arr;
  string txt;
  cin >> txt;

  for (int i = 0; i < txt.length(); i++) {
    int j = 0;
    while (j < arr.size() && arr[j] != txt[i]) {
      j++;
    }

    if (j == arr.size()) {
      arr.push_back(txt[i]);
    }
  }

  arr.size() % 2 == 0 ? cout << "CHAT WITH HER!" : cout << "IGNORE HIM!";

  return 0;
}
