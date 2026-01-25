#include <iostream>
#include <map>
#include <string>

using namespace std;
int get_distinct_char_count(const string &);
bool is_female(const string &);

int main(void) {
  string username;
  cin >> username;

  if (is_female(username))
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}

int get_distinct_char_count(const string &str) {
  map<char, int> char_frequency;

  for (char ch : str)
    if (!char_frequency.count(ch))
      char_frequency.insert(make_pair(ch, 1));
    else
      char_frequency[ch]++;

  return char_frequency.size();
}

bool is_female(const string &username) {
  if (get_distinct_char_count(username) % 2 == 0)
    return true;
  else
    return false;
}