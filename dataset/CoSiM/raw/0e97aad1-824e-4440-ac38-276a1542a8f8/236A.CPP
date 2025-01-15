#include <iostream>
#include <string>
#include <vector>

using namespace std;

short is_contain(vector<char> distinct_letters, char check) {

  for (short i = 0; i < distinct_letters.size(); i++) {
    if (distinct_letters[i] == check) {
      return 1;
    }
  }

  return 0;
}

int main() {
  string user_name;
  cin >> user_name;

  vector<char> distinct_letters{};
  for (short i = 0; i < user_name.size(); i++) {

    if (is_contain(distinct_letters, user_name[i]) == 0) {
      distinct_letters.push_back(user_name[i]);
    }
  }

  if (distinct_letters.size() % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
}