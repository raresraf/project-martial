#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;

int main() {
  string n;
  cin >> n;
  string string1 = "1111111";
  string string2 = "0000000";
  string answer = "NO";
  size_t pos = n.find(string1);
  size_t pos2 = n.find(string2);

  if (pos != string::npos) {
    cout << "YES";

  } else if (pos2 != string::npos) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
