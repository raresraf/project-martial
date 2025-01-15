#include <iostream>
#include <string>
#include <unordered_set>

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  string name;
  cin >> name;
  std::unordered_set<char> nameUsed;
  int distinct = 0;
  for (auto chr : name) {
    if (nameUsed.find(chr) == nameUsed.end()) {
      distinct++;
    }
    nameUsed.insert(chr);
  }

  if (distinct % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  return 0;
}
