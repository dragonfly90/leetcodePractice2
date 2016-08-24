//
//  main.cpp
//  leetcodePracticeRush
//
//  Created by Liang Dong on 7/26/16.
//  Copyright © 2016 Liang Dong. All rights reserved.
//

#include<iostream>
#include<vector>
#include<set>
#include<unordered_map>

using namespace std;

class SolutionmaxSumSubmatrix {
    
    int maxs;
    
public:
    int maxSumSubmatrix(vector<vector<int>>& matrix, int k) {
        /*
        int m=(int)matrix.size();
        if(m<=0)
            return 0;
        int n=(int)matrix[0].size();
        if(n<=0)
            return 0;
        
        maxs=INT_MIN;
        
        vector<vector<int> > allmatrixsum(m,vector<int>(n,0));
        int area;
        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++){
                area=matrix[i][j];
                if(i-1>=0)
                    area+=allmatrixsum[i-1][j];
                if(j-1>=0)
                    area+=allmatrixsum[i][j-1];
                if(i-1>=0&&j-1>=0)
                    area-=allmatrixsum[i-1][j-1];
                
                allmatrixsum[i][j]=area;
            }
        
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                for(int f=i;f<m;f++)
                {
                    for(int l=j;l<n;l++)
                    {
                        area=allmatrixsum[f][l];
                        if(i-1>=0)
                            area-=allmatrixsum[i-1][l];
                        if(j-1>=0)
                            area-=allmatrixsum[f][j-1];
                        if(i-1>=0&&j-1>=0)
                            area+=allmatrixsum[i-1][j-1];
                        if(area<=k)
                            maxs=max(maxs,area);
                    }
                    
                }
                
            }
        }
        
        return maxs;
         */
        if (matrix.empty()) return 0;
        int row = (int)matrix.size();
        int col =(int) matrix[0].size();
        int res = INT_MIN;
        for (int i = 0; i < col; i++) {
            vector<int> sums(row, 0);
            for (int r = i; r < col; r++) {
                for (int k = 0; k < row; k++) {
                    sums[k] += matrix[k][r];
                }
                
                
                set<int> accuSet;
                accuSet.insert(0);
                int curSum = 0, curMax = INT_MIN;
                for (int sum : sums) {
                    curSum += sum;
                    set<int>::iterator it = accuSet.lower_bound(curSum - k);
                    if (it != accuSet.end()) curMax = std::max(curMax, curSum - *it);
                    accuSet.insert(curSum);
                }
                res = std::max(res, curMax);
            }
        }
        return res;
    }
    
};

int best_cumulative_sum(vector<int> ar,int N,int K)
{
    set<int> cumset;
    cumset.insert(0);
    int best=0,cum=0;
    for(int i=0;i<N;i++)
    {
        cum+=ar[i];
        set<int>::iterator sit=cumset.upper_bound(cum-K);
        
        if(sit!=cumset.end())
            best=max(best,cum-*sit);
        
        cumset.insert(cum);
    }
    return best;
}

class SolutionminSubArrayLen {
    public:
        int minSubArrayLen(int s, vector<int>& nums) {
            /*
            sort(nums.begin(),nums.end(),greater<int>());
            int sum=0;
            vector<int> current;
            for(int i=0;i<nums.size();i++)
            {
                sum+=nums[i];
                if(sum>s)
                    return i+1;
                
            }
            
            return 0;
             */
            if(nums.size()<=0)
                return 0;
            int i=0,j=0,sum=0,minv=INT_MAX;
            while(j<nums.size()){
                sum+=nums[j++];
                while(sum>=s){
                    minv=min(minv,j-i);
                    sum-=nums[i++];
                }
            }
            
            return minv==INT_MAX?0:minv;
        }
};

class SolutioncountRangeSum {
public:
    int countRangeSum(vector<int>& nums, int lower, int upper) {
        int allcounts=0;
        int nsize=(int)nums.size();
        vector<vector<int> > suma(nsize,vector<int>(nsize,0));
        for(int i=0;i<nsize;i++)
            for(int j=i;j<nsize;j++)
            {
                if(j==i)
                    suma[i][j]=nums[i];
                else
                    suma[i][j]=nums[j]+suma[i][j-1];
                
                if(suma[i][j]>=lower&&suma[i][j]<=upper)
                    allcounts++;
            }
        
        return allcounts;
    }
};

//Definition for singly-linked list.
struct ListNode {
        int val;
        ListNode *next;
        ListNode(int x) : val(x), next(NULL) {}
};


class SolutiongetRandom {
    ListNode* currentHead;
    long long length;
public:
    /** @param head The linked list's head.
     Note that the head is guaranteed to be not null, so it contains at least one node. */
    SolutiongetRandom(ListNode* head) {
        length=0;
        currentHead=head;
        ListNode * temp=head;
        while(temp!=NULL)
        {
            length+=1;
            temp=temp->next;
        }
        
    }
    
    /** Returns a random node's value. */
    int getRandom() {
        ListNode * temp=currentHead;
        
        long long currenl=length;
        while(temp!=NULL)
        {
            if((double)rand()/(RAND_MAX)<1/double(currenl))
                return temp->val;
            else
                currenl-=1;
            temp=temp->next;
        }
        return 0;
        
    }
};


class SolutionShuffle{
    vector<int> shuffleVector;
    int nSize;
public:
    SolutionShuffle(vector<int> nums) {
        shuffleVector.assign(nums.begin(),nums.end());
        nSize=(int)shuffleVector.size();
    }
    
    /** Resets the array to its original configuration and return it. */
    vector<int> reset() {
        return shuffleVector;
    }
    
    /** Returns a random shuffling of the array. */
    vector<int> shuffle() {
        vector<int> returnvector;
        returnvector.assign(shuffleVector.begin(),shuffleVector.end());
        for(int j = 1; j < nSize; j++) {
            int i = rand()%(j+1);
            int temp=returnvector[i];
            returnvector[i]=returnvector[j];
            returnvector[j]=temp;
        }
        return returnvector;
        
    }
    
};

class SolutionremoveInvalidParentheses {
    
public:
    vector<string> removeInvalidParentheses(string s) {
        vector<string> allstrs;
        set<string> newallstrs;
        vector<string> tempstrs;
        set<string> strset;
        allstrs.clear();
        allstrs.push_back(s);
        
        while(1)
        {
            strset.clear();
            bool valid=true;
            for(auto tempstr:allstrs)
            {
                if(isvalid(tempstr))
                {
                    strset.insert(tempstr);
                }
                else
                    valid=false;
            }
            
            
            if(!strset.empty())
            {
                return vector<string>(strset.begin(),strset.end());
            }
            
            newallstrs.clear();
            for(auto tempstr:allstrs)
            {
                for(int i=1;i<tempstr.size();i++)
                    newallstrs.insert(tempstr.substr(0,i)+tempstr.substr(i+1));
                newallstrs.insert(tempstr.substr(1));
            }
            //for(auto i:newallstrs)
            //    cout<<i<<endl;
            allstrs.clear();
            allstrs.assign(newallstrs.begin(),newallstrs.end());
            
        }
    }
    
    bool isvalid(string s){
        int ctr=0;
        for(auto i:s)
        {
            if(i=='(')
                ctr+=1;
            else
                if(i==')')
                    ctr-=1;
            if(ctr<0)
                return false;
        }
        
        return ctr==0;
    }
    

    
};

class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        unordered_map<int,int> allset;
        
        for(int i=0;i<numbers.size();i++)
        {
            
            if(allset.find(numbers[i])!=allset.end())
            {
                vector<int> returnIndex;
                returnIndex.push_back(allset[numbers[i]]);
                returnIndex.push_back(i);
                return returnIndex;
            }
            else
                allset[target-numbers[i]]=i;
        }
        return vector<int>();
    }
    
    vector<int> twoSum2(vector<int>& numbers, int target){
        int lo=0, hi=(int)numbers.size()-1;
        while (numbers[lo]+numbers[hi]!=target){
            if (numbers[lo]+numbers[hi]<target){
                lo++;
            } else {
                hi--;
            }
        }
        return vector<int>({lo+1,hi+1});
    }
};

int main(){
    /*
    SolutionmaxSumSubmatrix solution;
    vector<vector<int> > testmatrix;
    int k=2;
    int matrix1[]={1,0,1};
    int matrix2[]={0,-2,3};
    testmatrix.resize(2);
    testmatrix[0].assign(matrix1,matrix1+2);
    testmatrix[1].assign(matrix2,matrix2+2);
    
    cout<<solution.maxSumSubmatrix(testmatrix, k)<<endl;
     */
    SolutionremoveInvalidParentheses mysolution;
    vector<string> allstrs=mysolution.removeInvalidParentheses("()())()");
    for(auto str: allstrs)
       cout<<str<<endl;
    
}
    
